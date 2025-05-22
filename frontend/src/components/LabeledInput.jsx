import React from 'react'

const LabeledInput = ({ label, type = 'text', name, value, onChange, classname = '' }) => {
    return (
        <div className='w-full'>
            <label
                htmlFor={name}
                className="block font-medium mb-2">
                {label}
            </label>
            <input
                type={type}
                name={name}
                value={value}
                className={`border border-gray-500 rounded-2xl p-2 pl-4 w-full ${classname}`}
                onChange={onChange}
            />
        </div>
    )
}

export default LabeledInput